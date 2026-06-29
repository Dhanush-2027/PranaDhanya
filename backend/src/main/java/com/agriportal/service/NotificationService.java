package com.agriportal.service;

import com.agriportal.entity.Notification;
import com.agriportal.entity.User;
import com.agriportal.repository.NotificationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@SuppressWarnings("null")
public class NotificationService {

    @Autowired
    private NotificationRepository notificationRepo;

    @Transactional
    public Notification createNotification(User user, String title, String message, String type) {
        Notification notification = new Notification(user, title, message, type);
        return notificationRepo.save(notification);
    }

    @Transactional
    public Notification createGlobalNotification(String title, String message, String type) {
        Notification notification = new Notification(null, title, message, type);
        return notificationRepo.save(notification);
    }

    public List<Notification> getNotificationsForUser(Long userId) {
        return notificationRepo.findByUserIdOrderByTimestampDesc(userId);
    }

    public List<Notification> getUnreadNotificationsForUser(Long userId) {
        return notificationRepo.findByUserIdAndReadOrderByTimestampDesc(userId, false);
    }

    public List<Notification> getGlobalNotifications() {
        return notificationRepo.findByUserIdIsNullOrderByTimestampDesc();
    }

    @Transactional
    public void markAsRead(Long notificationId) {
        notificationRepo.findById(notificationId).ifPresent(n -> {
            n.setRead(true);
            notificationRepo.save(n);
        });
    }

    @Transactional
    public void deleteNotification(Long id) {
        notificationRepo.deleteById(id);
    }
}
